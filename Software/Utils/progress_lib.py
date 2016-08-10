# !/usr/bin/python
# -*- coding: utf8 -*-
#                         -.
#                  ,,,,,,    `²╗,
#             ╓▄▌█▀▀▀▀▀▀▀█▓▓▓▓▌▄  ╙@╕
#          ▄█▀Γ,╓╤╗Q╣╣╣Q@╤═ Γ▀▓▓▓▓▄ "▒╦
#        ▄▀,╤╣▒▒▒▒▒▒ÅÅ╨╨╨ÅÅ▒▒▒╤▐▓▓▓▓▄ ╙▒╕ └
#      4Γ,╣▒▒ÅÖ▄▓▓▓▓▓█%─     `Å▒Q█▓▓▓▓ └▒╦ ▐╕
#       ╩▒▒`╓▓▓▓▓▀Γ             ╙▒▀▓▓▓▓ ╚▒╕ █
#      ▒▒ ,▓▓▓▓Γ ,                ì▀▓▓▓▌ ▒▒  ▓
#     ▒▒ ╓▓▓▓▀,Q▒                   ▓▓▓▓ ▒▒⌐ ▓
#    ╓▒ ╒▓▓▓▌╣▒▒                    ▓▓▓▓║▒▒⌐ ▓─
#    ╫Γ ▓▓▓█▒▒▒∩                    ▓▓▓▌▒▒▒ ]▓
#    ╫⌐ ▓▓▓]▒▒▒                    ▓▓▓Θ▒▒▒O ▓▓
#    ║µ ▓▓▌ ▒▒▒╕                 ,█▀Γ╒▒▒▒┘ ▓▓`
#     Θ ▀▓▓ ▒▒▒▒⌐▄                 ,╣▒▒Å ▄▓▓Γ
#     ╚  ▓▓ '▒▒▒▒▓▓▄           ,═Q▒▒▒Ö,▄▓▓█ .
#      ╙  ▓▓ "▒▒▒▒╬█▓▓▄▄     `╙╨╢▄▓▓▓▓▓▓█Γ╒┘
#          ▀▓▄ Å▒▒▒▒ç▀█▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▀,d┘
#            ▀▓▄ ╙▒▒▒▒╗, Γ▀▀▀▀▀▀▀Γ ,╓ê╜
#              ▀█▄▄  ╙ÅÅ▒▒▒╣QQQ╩ÅÅ╙
#                  ▀▀m▄
#
#
__author__ = 'pablogsal'
#--------------------------------------IMPORT STUFF----------------------------------------------#

import os
import sys

#--------------------------------------FUNCTION DECLARATION--------------------------------------#

# getTerminalSize() : Returns the width and the height of the terminal as a tuple.
def getTerminalSize():
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct, os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
        '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        ### Use get(key[, default]) instead of a try/catch
        #try:
        #    cr = (env['LINES'], env['COLUMNS'])
        #except:
        #    cr = (25, 80)
    return int(cr[1]), int(cr[0])


# update_progress() : Displays or updates a console progress bar
# Accepts a float between 0 and 1. Any int will be converted to a float.
# A value under 0 represents a 'halt'.
# A value at 1 or bigger represents 100%
# The width of the progress bar is adapted to the console width and is updated in every iteration of the code.
def update_progress(progress, time):
    (width, height) = getTerminalSize()
    barLength = width/2  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1:.2f}% --- {3:.2f} s. remain. {2}".format(
        "=" * (block - 1) + ">" + " " * int(barLength - (block - 1)-1), progress * 100, status, time)
    sys.stdout.write(text)
    sys.stdout.flush()


# Dummy -and probably inefficient- implementation of the mean
def mean(vector):
    return float(sum(vector) / len(vector))


def progress_bar_init(max):
    meanTime = []
    step = 0

    def update(lastStepTime):
        nonlocal step
        # Append time measure to the measure list
        meanTime.append(lastStepTime)

        # Update the progress bar
        update_progress(
            float(step) / max, mean(meanTime) * (max - step)
        )

        step += 1

    return update
