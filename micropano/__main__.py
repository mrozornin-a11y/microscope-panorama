import sys

if len(sys.argv) > 1 and sys.argv[1] != "--gui":
    from .cli import main
else:
    from .gui import main

sys.exit(main())
