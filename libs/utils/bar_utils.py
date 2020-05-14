import progressbar

def create_bar(begin_info, len):
    widgets = [begin_info, progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.Counter(), ' / %s' % len,
                    ' ', progressbar.ETA(), ' ']
    bar = progressbar.ProgressBar(maxval=len, widgets=widgets)

    return bar

if __name__ == "__main__":
    import time
    info = 'Test_info_%d'%100
    bar = create_bar(info, 100)
    bar.start()
    for i in range(100):
        bar.update()
        time.sleep(0.2)

    bar.finish()