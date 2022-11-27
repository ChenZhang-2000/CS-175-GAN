import time
import datetime


def progress_bar(current, total, bar_length=20, progress_name="Progress", t=0):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'{progress_name}: [{arrow}{padding}] {int(fraction*100)}%  Time Spent: {datetime.timedelta(seconds=time.time()-t)}', end=ending)
