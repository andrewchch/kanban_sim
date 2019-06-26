class Release(object):
    def __init__(self, env=None):
        self._in_test = False
        self.released = False
        self.released_time = None
        self.cases = []
        self.env = env

    @property
    def in_test(self):
        return self._in_test

    @in_test.setter
    def in_test(self, val):
        self._in_test = val

    def release(self):
        self._in_test = False
        self.released = True
        self.released_time = self.env.now
        for _case in self.cases:
            _case.do_release()

    def add_case(self, case):
        self.cases.append(case)
        case.release = self

    def __str__(self):
        return 'Release with %d cases, in_test = %s, released_time = %s' % (len(self.cases), self._in_test, self.released_time)


