import time

def train_print(loss, step, total_step, epoch, step_time, epoch_time):
    epoch_time = time.localtime(epoch_time)
    min = epoch_time.tm_min
    sec = epoch_time.tm_sec
    print(f"\rloss:{format(loss, '.2f')} |"
          f"step: {step}/{total_step} |"
          f"epoch: {epoch} |"
          f"step time:{format(step_time, '.2f')}secs |",
          f"epoch time: {min}min {sec}sec", end='')