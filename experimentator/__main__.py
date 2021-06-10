import os
from dotenv import load_dotenv

load_dotenv() # For weird reason, __init__ is called before and putting 'load_dotenv()' here has not the desired effect of setting 'TF_CPP_MIN_LOG_LEVEL' before the import of main

from .manager import main # pylint: disable=wrong-import-position

if __name__ == "__main__":
    # Add current working directory to PATH
    if not os.getcwd() in os.sys.path:
        os.sys.path.insert(0, os.getcwd())

    print(os.uname().nodename.upper().lower(), os.sys.executable)

    main()
