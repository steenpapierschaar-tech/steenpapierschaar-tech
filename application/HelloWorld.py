# Hello world in Python

import platform


if __name__ == '__main__':
    # Get the operating system name
    os_name = platform.system()

    # Get the system's architecture
    architecture = platform.architecture()

    print(f"Hello Wordl! from {os_name} operating system on {architecture[0]} {architecture[1]} architecture")
else:
    pass