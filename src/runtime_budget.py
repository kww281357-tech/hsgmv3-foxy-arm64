"""Navigation budgets, independent of Habitat. Resource exhaustion is not STOP."""
import math
import signal
import threading
import time
from contextlib import contextmanager


class EpisodeBudgetExceeded(BaseException):
    """Bypass planner fallbacks that catch Exception and otherwise keep moving."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class EpisodeBudget:
    def __init__(self, seconds=900, max_requests=48, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.seconds = float(seconds)
        self.max_requests = int(max_requests)
        self.requests = 0
        if not math.isfinite(self.seconds) or self.seconds < 0 or self.max_requests < 0:
            raise ValueError('Budgets must be finite and non-negative; zero disables a limit.')

    def remaining(self):
        return max(0.0, self.seconds - (self.clock() - self.started)) if self.seconds else math.inf

    def check(self):
        if self.remaining() <= 0:
            raise EpisodeBudgetExceeded('wall_time')

    def claim_request(self):
        self.check()
        if self.max_requests and self.requests >= self.max_requests:
            raise EpisodeBudgetExceeded('model_requests')
        self.requests += 1

    def summary(self):
        return {'elapsed_s': round(self.clock() - self.started, 3),
                'limit_s': self.seconds, 'model_attempts': self.requests,
                'model_attempt_limit': self.max_requests}

    @contextmanager
    def guard(self):
        """Interrupt Python/socket waits on Linux; native GPU kernels may defer signals.

        Refuse to replace another owner's timer. Cooperative checks remain in
        force on other threads/platforms. No background worker survives timeout.
        """
        self.check()
        supported = hasattr(signal, 'setitimer') and threading.current_thread() is threading.main_thread()
        armed = supported and math.isfinite(self.remaining()) and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        old_handler = None
        if armed:
            old_handler = signal.getsignal(signal.SIGALRM)
            def expire(signum, frame):
                raise EpisodeBudgetExceeded('wall_time')
            signal.signal(signal.SIGALRM, expire)
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, self.remaining()))
        try:
            yield
            self.check()
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

