import time, os, sys

if __name__ == '__main__':
    # Get the first argument
    arg1 = sys.argv[1]

    # create dir with current date and time
    dir_name = f"cluster_{arg1}"
    os.makedirs(dir_name)

    for n in os.listdir():
        # if it is a directory and dir name is a number
        if os.path.isdir(n) and n.isdigit():
            # Move it to the new dir
            os.rename(n, os.path.join(dir_name, n))
        # Move the testingSetScores.json file to the new dir
        if n == "testingSetScores.json":
            os.rename(n, os.path.join(dir_name, n))

    # tar the dir
    tar_name = f"{dir_name}.tar.gz"
    os.system(f"tar -czf {tar_name} {dir_name}")

    # remove the dir
    os.system(f"rm -rf {dir_name}")

    # move tar file to /staging/djpaul2
    staging_dir = "/staging/djpaul2"
    if os.path.exists(staging_dir):
        os.system(f"mv {tar_name} {staging_dir}")