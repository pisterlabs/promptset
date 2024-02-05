import sys

if __name__ == '__main__':
    assert len(sys.argv) == 2
    file = sys.argv[1]
    print('file:', file)
    sys.stdout.flush()
