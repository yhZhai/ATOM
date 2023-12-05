import os
import time
from types import SimpleNamespace
import blobfile as bf

import numpy as np
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from eval import eval_humanml_cvae, eval_humanact12_uestc_cvae
from data_loaders.get_data import get_dataset_loader
from losses.losses import Losses
from utils.misc import MetricLogger

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        self.num_steps = int(self.num_epochs * len(self.data))

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, int(self.num_steps * 3 / 4), gamma=0.1)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        # self.schedule_sampler_type = "uniform"
        # self.schedule_sampler = create_named_schedule_sampler(
        #     self.schedule_sampler_type, diffusion
        # )
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ["kit", "humanml"] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(
                name=args.dataset,
                batch_size=args.eval_batch_size,
                num_frames=None,
                split=args.eval_split,
                hml_mode="eval",
            )

            self.eval_gt_data = get_dataset_loader(
                name=args.dataset,
                batch_size=args.eval_batch_size,
                num_frames=None,
                split=args.eval_split,
                hml_mode="gt",
            )
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                "test": lambda: eval_humanml_cvae.get_cvae_loader(
                    model,
                    args.eval_batch_size,
                    gen_loader,
                    mm_num_samples,
                    mm_num_repeats,
                    gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples,
                    scale=1.0,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model
        self.losses = Losses().to(self.device)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch} / {self.num_epochs}")
            metric_logger = MetricLogger()
            # for motion, cond in tqdm(self.data):
            for i, (motion, cond) in metric_logger.log_every(
                self.data, print_freq=self.args.print_freq
            ):

                # b, #keypoints, 1, T
                # T is masked to the longest sequence in the batch
                motion = motion.to(self.device)
                cond["y"] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond["y"].items()
                }
                cond["mask_ratio"] = self.args.mask_ratio

                losses = self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k, v in logger.get_current().name2val.items():
                        if k == "loss":
                            print(
                                "step[{}]: loss[{:0.5f}]".format(
                                    self.step + self.resume_step, v
                                )
                            )

                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k, value=v, iteration=self.step, group_name="Loss"
                            )

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
                metric_logger.update(**losses)
            print("Average status:")
            print(metric_logger.stat_table())

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print("Running evaluation loop: [Should take about 90 min]")
            log_file = os.path.join(
                self.save_dir, f"eval_humanml_{(self.step + self.resume_step):09d}.log"
            )
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml_cvae.evaluation(
                self.eval_wrapper,
                self.eval_gt_data,
                self.eval_data,
                log_file,
                replication_times=self.args.eval_rep_times,
                diversity_times=diversity_times,
                mm_num_times=mm_num_times,
                run_mm=False,
            )
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith("R_precision"):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(
                            name=f"top{i + 1}_" + k,
                            value=v[i],
                            iteration=self.step + self.resume_step,
                            group_name="Eval",
                        )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=v,
                        iteration=self.step + self.resume_step,
                        group_name="Eval",
                    )

        # TODO not implemented
        elif self.dataset in ["humanact12", "uestc"]:
            eval_args = SimpleNamespace(
                num_seeds=self.args.eval_rep_times,
                num_samples=self.args.eval_num_samples,
                batch_size=self.args.eval_batch_size,
                device=self.device,
                guidance_param=1,
                dataset=self.dataset,
                unconstrained=self.args.unconstrained,
                model_path=os.path.join(self.save_dir, self.ckpt_file_name()),
                cond_mode=self.model.cond_mode,
            )
            eval_dict = eval_humanact12_uestc_cvae.evaluate(
                eval_args,
                model=self.model,
                data=self.data.dataset,
            )
            print(
                f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}'
            )
            for k, v in eval_dict["feats"].items():
                if "unconstrained" not in k:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval",
                    )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval Unconstrained",
                    )

        end_eval = time.time()
        print(f"Evaluation time: {round(end_eval-start_eval)/60}min")

    def run_step(self, batch, cond):
        losses = self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.lr_scheduler.step()
        self.log_step()
        return losses

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        micro = batch
        micro_cond = cond
        last_batch = self.microbatch >= batch.shape[0]
        # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

        # compute_losses = functools.partial(
        #     self.diffusion.training_losses,
        #     self.ddp_model,
        #     micro,  # [bs, ch, image_size, image_size]
        #     t,  # [bs](int) sampled timesteps
        #     model_kwargs=micro_cond,
        #     dataset=self.data.dataset,
        # )

        if last_batch or not self.use_ddp:
            losses = self.compute_losses(
                self.model, micro, micro_cond, self.data.dataset
            )
        else:
            with self.ddp_model.no_sync():
                losses = self.compute_losses(
                    self.model, micro, micro_cond, self.data.dataset
                )

        # if isinstance(self.schedule_sampler, LossAwareSampler):
        #     self.schedule_sampler.update_with_local_losses(
        #         t, losses["loss"].detach()
        #     )

        loss = losses["total_loss"]
        # log_loss_dict(
        #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
        # )
        self.mp_trainer.backward(loss)
        return losses

    def compute_losses(self, model, x, model_kwargs, dataset):
        mask = model_kwargs["y"]["mask"]
        get_xyz = lambda sample: model.rot2xyz(
            sample,
            mask=None,
            pose_rep=model.pose_rep,
            translation=model.translation,
            glob=model.glob,
            # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
            jointstype="smpl",  # 3.4 iter/sec
            vertstrans=False,
        )

        if model_kwargs is None:
            model_kwargs = {}

        num_step_before_anneal = self.num_steps * 3 / 4
        if self.args.mask_sched == "constant":
            mask_ratio = self.args.mask_ratio
        elif self.args.mask_sched == "linear":
            mask_ratio = self.args.mask_ratio * self.step / num_step_before_anneal
        elif self.args.mask_sched == "square":
            mask_ratio = self.args.mask_ratio * (self.step / num_step_before_anneal) ** 2
        elif self.args.mask_sched == "exp":
            mask_ratio = self.args.mask_ratio * np.exp(self.step / num_step_before_anneal - 1)
        model_kwargs["mask_ratio"] = mask_ratio

        model_output = model(x, **model_kwargs)

        # MSE loss on the rotation matrix
        rot_mse = self.losses.masked_l2(x, model_output["output"], mask)
        rot_mse = rot_mse.mean()

        # MSE loss on the keypoint coordinates
        if self.model.data_rep == "rot6d" and dataset.dataname in [
            "humanact12",
            "uestc",
        ]:
            target_xyz = get_xyz(x)
            model_output_xyz = get_xyz(model_output["output"])
            xyz_mse = self.losses.masked_l2(target_xyz, model_output_xyz, mask)
            xyz_mse = xyz_mse.mean()
        elif self.model.data_rep == "hml_vec":
            n_joints = 22 if x.shape[1] == 263 else 21
            std, mean = self.data.dataset.t2m_dataset.std, self.data.dataset.t2m_dataset.mean
            std, mean = torch.tensor(std, device=x.device), torch.tensor(mean, device=x.device)
            target_xyz = (x.permute(0, 2, 3, 1) * std + mean).float()
            target_xyz = recover_from_ric(target_xyz, n_joints)
            target_xyz = target_xyz.view(-1, *target_xyz.shape[2:]).permute(0, 2, 3, 1)
            model_output_xyz = (model_output["output"].permute(0, 2, 3, 1) * std + mean).float()
            model_output_xyz = recover_from_ric(model_output_xyz, n_joints)
            model_output_xyz = model_output_xyz.view(-1, *model_output_xyz.shape[2:]).permute(0, 2, 3, 1)
            xyz_mse = self.losses.masked_l2(target_xyz, model_output_xyz, mask)
            xyz_mse = xyz_mse.mean()
        else:
            xyz_mse = 0.0

        # KLD loss
        kl_loss = self.losses.kl_loss(model_output["mu"], model_output["logvar"])

        # attention sparsity loss
        att_spa_loss = self.losses.att_spa_loss(model_output["att"])

        # codebook norm loss
        codebook_norm_loss = self.losses.codebook_norm_loss(model_output["codebook"])

        total_loss = (
            self.args.rot_mse_w * rot_mse
            + self.args.xyz_mse_w * xyz_mse
            + self.args.kld_w * kl_loss
            + self.args.att_spa_w * att_spa_loss
            + self.args.codebook_norm_w * codebook_norm_loss
        )

        losses = {
            "total_loss": total_loss,
            "rot_mse": rot_mse,
            "xyz_mse": xyz_mse,
            "kl_loss": kl_loss,
            "att_spa_loss": att_spa_loss,
            "codebook_norm_loss": codebook_norm_loss,
            "mask_ratio": mask_ratio,
        }
        return losses

    def _anneal_lr(self):
        self.lr_scheduler.step()

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith("clip_model.")]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
