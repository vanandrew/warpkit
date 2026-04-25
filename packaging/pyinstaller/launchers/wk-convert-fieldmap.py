import multiprocessing

multiprocessing.freeze_support()

from warpkit.scripts.convert_fieldmap import main  # noqa: E402

main()
