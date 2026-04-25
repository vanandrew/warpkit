import multiprocessing

multiprocessing.freeze_support()

from warpkit.scripts.unwrap_phase import main  # noqa: E402

main()
