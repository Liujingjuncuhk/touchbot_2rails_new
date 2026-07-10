import multiprocessing
import time
import random

# ── Simulated hardware functions (stand-ins for real sensor/motor calls) ──

def fake_read_sensor():
    time.sleep(0.02)  # simulate 20ms Modbus read
    return [random.uniform(0.0, 1.0) for _ in range(4)]

def fake_send_motor_cmd(base_mm, vert_mm):
    time.sleep(0.05)  # simulate 50ms motor command
    return f"cmd sent: base={base_mm:.1f} vert={vert_mm:.1f}"


# ── Worker functions (must be top-level for multiprocessing to pickle them) ──

def sensor_worker(result_queue, stop_event, read_interval=0.05):
    """Continuously reads sensor and puts latest reading into the queue."""
    while not stop_event.is_set():
        forces = fake_read_sensor()
        # drain old values so queue never grows unboundedly
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                break
        result_queue.put(forces)
        time.sleep(read_interval)


def motor_worker(cmd_queue, log_queue, stop_event):
    """Consumes motor commands from cmd_queue and executes them."""
    while not stop_event.is_set():
        try:
            cmd = cmd_queue.get(timeout=0.1)
            result = fake_send_motor_cmd(cmd[0], cmd[1])
            log_queue.put(result)
        except Exception:
            pass  # timeout — just loop


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sensor_queue = multiprocessing.Queue()  # sensor_worker -> main
    cmd_queue    = multiprocessing.Queue()  # main -> motor_worker
    log_queue    = multiprocessing.Queue()  # motor_worker -> main
    stop_event   = multiprocessing.Event()

    # start parallel processes
    p_sensor = multiprocessing.Process(target=sensor_worker,
                                       args=(sensor_queue, stop_event, 0.05))
    p_motor  = multiprocessing.Process(target=motor_worker,
                                       args=(cmd_queue, log_queue, stop_event))
    p_sensor.start()
    p_motor.start()

    print("Processes started. Running for 2 seconds...\n")

    # ── Main loop: send motor commands and read latest sensor data ──
    t_start = time.time()
    cmd_base = 107.2
    step = 5.0

    while time.time() - t_start < 2.0:
        # send a motor command
        cmd_queue.put([cmd_base, 100.0])
        cmd_base += step

        # read latest sensor (non-blocking)
        forces = None
        if not sensor_queue.empty():
            forces = sensor_queue.get_nowait()

        # read motor log (non-blocking)
        log = None
        if not log_queue.empty():
            log = log_queue.get_nowait()

        print(f"t={time.time()-t_start:.2f}s | "
              f"forces={[round(f,3) for f in forces] if forces else 'pending'} | "
              f"motor: {log if log else 'pending'}")

        time.sleep(0.1)

    # ── Shutdown ──
    stop_event.set()
    p_sensor.join()
    p_motor.join()
    print("\nAll processes stopped.")
