"""Corrected READ-hidden/FM8 runtime; retain genuine empty curve outputs."""
import s3_grid_diagnostic as common

original_finish = common.PCM.finish


def finish(self, *args, **kwargs):
    kwargs['allow_empty'] = True
    return original_finish(self, *args, **kwargs)


common.PCM.finish = finish
if __name__ == '__main__':
    common.m.main()
