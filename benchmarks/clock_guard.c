/* Keep CLOCK_REALTIME monotonic for duration measurements in Docker Desktop/WSL2. */

#define _GNU_SOURCE

#include <pthread.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static struct timespec realtime_origin;
static struct timespec monotonic_origin;

static int raw_clock_gettime(clockid_t clock_id, struct timespec *value) {
    return (int)syscall(SYS_clock_gettime, clock_id, value);
}

static void initialize_origins(void) {
    raw_clock_gettime(CLOCK_REALTIME, &realtime_origin);
    raw_clock_gettime(CLOCK_MONOTONIC, &monotonic_origin);
}

static struct timespec add_elapsed(
    const struct timespec origin,
    const struct timespec start,
    const struct timespec current
) {
    struct timespec result;
    int64_t seconds = current.tv_sec - start.tv_sec;
    int64_t nanoseconds = current.tv_nsec - start.tv_nsec;
    if (nanoseconds < 0) {
        seconds -= 1;
        nanoseconds += 1000000000L;
    }
    result.tv_sec = origin.tv_sec + seconds;
    result.tv_nsec = origin.tv_nsec + nanoseconds;
    if (result.tv_nsec >= 1000000000L) {
        result.tv_sec += 1;
        result.tv_nsec -= 1000000000L;
    }
    return result;
}

int clock_gettime(clockid_t clock_id, struct timespec *value) {
    struct timespec monotonic_now;
    if (clock_id != CLOCK_REALTIME) {
        return raw_clock_gettime(clock_id, value);
    }
    pthread_once(&initialized, initialize_origins);
    if (raw_clock_gettime(CLOCK_MONOTONIC, &monotonic_now) != 0) {
        return -1;
    }
    *value = add_elapsed(realtime_origin, monotonic_origin, monotonic_now);
    return 0;
}

int gettimeofday(struct timeval *value, void *timezone) {
    struct timespec realtime_now;
    (void)timezone;
    if (value == NULL || clock_gettime(CLOCK_REALTIME, &realtime_now) != 0) {
        return -1;
    }
    value->tv_sec = realtime_now.tv_sec;
    value->tv_usec = (suseconds_t)(realtime_now.tv_nsec / 1000L);
    return 0;
}
