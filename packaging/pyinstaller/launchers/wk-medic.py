import multiprocessing

multiprocessing.freeze_support()

from warpkit.scripts.medic import main  # noqa: E402

main()
