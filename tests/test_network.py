from ecmgen.networktype import NetworkType

import unittest

class TestStringMethods(unittest.TestCase):

    def test_creation(self):
        network_type = NetworkType(
            None, None, None
        )
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()