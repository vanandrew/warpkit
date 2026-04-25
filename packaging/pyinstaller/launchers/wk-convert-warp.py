import multiprocessing

multiprocessing.freeze_support()

from warpkit.scripts.convert_warp import main  # noqa: E402

main()
