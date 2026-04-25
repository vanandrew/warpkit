import multiprocessing

multiprocessing.freeze_support()

from warpkit.scripts.apply_warp import main  # noqa: E402

main()
