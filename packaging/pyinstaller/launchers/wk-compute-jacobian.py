import multiprocessing

multiprocessing.freeze_support()

from warpkit.scripts.compute_jacobian import main  # noqa: E402

main()
