from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


class CDCLRScheduler(Callback):
    """
    * Cosine Decay ,Warm-Up and Cycle Restart.
    * For Tensorflow 2 :)
    * Based on https://gist.github.com/jeremyjordan/5a222e04bb78c242f5763ad40626c452
    ---
    Ver. IB811_LIAO_2021_04_09
    """

    def __init__(
            self,
            base_lr,
            alpha=0.,
            initial_epoch=0,
            steps_per_epoch=None,
            cycle_length=2,
            factor=1,
            warm_up=False,
            warm_up_epochs=0,
            warm_up_lr_factor=0.1,
            save_txt=False,
            file_name=None,
    ):
        """
        :param base_lr: Base learning rate.
        :param alpha: Multiple factor for minimum learning rate.
        :param initial_epoch: .
        :param steps_per_epoch: .
        :param cycle_length: Restart cycle length. If you want only one cycle to train, set the same parameter as epochs.
        :param factor: Restart cycle length factor.
        :param warm_up: True for training with warm-up method.
        :param warm_up_epochs: Epochs to warm-up.
        :param warm_up_lr_factor: .
        :param save_txt: Save Learning rate and Losses as .txt.
        :param file_name: Text's file name.
        """
        super(CDCLRScheduler, self).__init__()
        self.base_lr = base_lr
        self.alpha = alpha
        self.initial_epoch = initial_epoch
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = cycle_length
        self.factor = factor
        self.warm_up = warm_up
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_lr = base_lr * warm_up_lr_factor
        self.save_txt = save_txt
        self.file_name = file_name

        self.total_steps = 0
        self.total_wp_steps = 0
        self.batch_since_restart = 0
        self.step = 0
        self.next_restart = cycle_length
        self.flag = 1

        self.learning_rates = []
        self.losses = []

    # Cosine Decay lr
    def clr(self):
        if self.warm_up and self.step == self.total_wp_steps:
            self.batch_since_restart = 0

        length = (self.steps_per_epoch * self.cycle_length)

        if self.warm_up and self.step > self.total_wp_steps and self.flag:
            if self.step < self.steps_per_epoch * self.cycle_length:
                length -= (self.steps_per_epoch * self.warm_up_epochs)
            else:
                self.flag = 0

        # Original
        fraction_to_restart = self.batch_since_restart / length
        decay = 0.5 * (1 + np.cos(fraction_to_restart * np.pi))
        decay = self.alpha + (1 - self.alpha) * decay
        lr = self.base_lr * decay
        return lr

    # Warm Up lr
    def wlr(self):
        decay = self.warm_up_lr + (self.base_lr - self.warm_up_lr) * (
                self.step / (self.steps_per_epoch * self.warm_up_epochs))
        return decay

    def check_restart(self, epoch):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.factor)
            self.next_restart += self.cycle_length

    def batch_count(self, num_steps):
        self.batch_since_restart += num_steps
        self.step += num_steps

    def on_train_begin(self, logs=None):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.params['steps']

        self.total_steps = self.params['epochs'] * self.params['steps']

        if self.warm_up:
            self.total_wp_steps = self.warm_up_epochs * self.steps_per_epoch

        print(f"[INFO] Epochs : {self.params['epochs']}")
        print(f"[INFO] Steps per epoch : {self.params['steps']}")
        print(f"[INFO] Total steps : {self.total_steps}")
        print(f"[INFO] Current Step : {self.step + 1}")

        # check for initial epoch
        if self.initial_epoch != 0:
            for i in range(self.initial_epoch):
                self.batch_count(num_steps=self.steps_per_epoch)
                self.check_restart(epoch=i)

        # if warm up is used, and initial epoch must be 0.
        if self.warm_up and self.initial_epoch == 0:
            K.set_value(self.model.optimizer.lr, self.wlr())

        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_begin(self, batch, logs=None):
        if self.warm_up and self.step < self.total_wp_steps:
            self.learning_rates.append(self.wlr())

        else:
            self.learning_rates.append(self.clr())

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get("loss"))
        self.batch_count(num_steps=1)

        if self.warm_up and self.step < self.total_wp_steps:
            K.set_value(self.model.optimizer.lr, self.wlr())

        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_begin(self, epoch, logs=None):
        if self.warm_up and self.step < self.total_wp_steps:
            print(f"[INFO] Steps {self.step + 1}/{self.total_wp_steps} | Warm Up Lr : {self.wlr()}")

        else:
            print(f"[INFO] Steps {self.step + 1}/{self.total_steps} | Cosine Decay Lr : {self.clr()}")

    def on_epoch_end(self, epoch, logs=None):
        self.check_restart(epoch=epoch)

    def on_train_end(self, logs=None):
        if self.save_txt:
            if self.save_txt and self.file_name is None:
                self.file_name = 'CDCLRScheduler'

            np.savetxt(f'{self.file_name}_lrs.txt', self.learning_rates)
            np.savetxt(f'{self.file_name}_losses.txt', self.losses)
