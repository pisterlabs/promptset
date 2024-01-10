#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from openai_helper.dmo import InputTokenCounter


def test_service():

    counter = InputTokenCounter().process
    assert counter

    print (counter('the quick brown fox jumps over the lazy dog'))


def main():
    test_service()


if __name__ == '__main__':
    main()
