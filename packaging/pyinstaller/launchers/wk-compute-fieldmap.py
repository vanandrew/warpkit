import multiprocessing

multiprocessing.freeze_support()

from warpkit.scripts.compute_fieldmap import main  # noqa: E402

main()
