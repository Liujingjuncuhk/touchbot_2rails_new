import time
import numpy as np
import matplotlib.pyplot as plt
from pymodbus.client import ModbusSerialClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian


PORT = '/dev/touchbot_touchForceSensor'
BAUDRATE = 115200
SLAVE_ID = 1
# Conversion ratios per channel (raw int -> physical units)
# RATIO = [1 / 7200, 1 / 11, 1 / 10000, 1 / 6900]
RATIO = [1/100, 1/100, 1/100, 1/100]

class TouchForceSensorArray:
    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        slave_id: int = SLAVE_ID,
    ):
        self.slave_id = slave_id
        self.ratio = RATIO
        self._client = ModbusSerialClient(port=port, baudrate=baudrate, timeout=0.2)
        if not self._client.connect():
            raise RuntimeError(f"Cannot open serial port {port}")
        self.calibrate_initial_offset()
        
    def calibrate_initial_offset(self, nSample = 10):
        total = [0] * 4
        for _ in range(nSample):
            forces = self.read_forces_raw()
            if forces:
                for i in range(4):
                    total[i] += forces[i]
        self.initial_offset = [t / nSample for t in total]

    def read_forces_raw(self) -> list:
        """Read all 4 channel force values.

        Returns:
            List of 4 floats [ch1, ch2, ch3, ch4] in physical units,
            or None if the read fails.
        """
        result = self._client.read_holding_registers(
            address=450, count=8, slave=self.slave_id
        )
        if result.isError():
            return None

        decoder = BinaryPayloadDecoder.fromRegisters(
            result.registers,
            byteorder=Endian.BIG,
            wordorder=Endian.BIG,
        )
        raw = [decoder.decode_32bit_int() for _ in range(4)]
        returned_force = [raw[i] * self.ratio[i] for i in range(4)]
        return returned_force

    def read_forces(self) -> list:
        """Read all 4 channel force values.

        Returns:
            List of 4 floats [ch1, ch2, ch3, ch4] in physical units,
            or None if the read fails.
        """
        result = self._client.read_holding_registers(
            address=450, count=8, slave=self.slave_id
        )
        if result.isError():
            return None

        decoder = BinaryPayloadDecoder.fromRegisters(
            result.registers,
            byteorder=Endian.BIG,
            wordorder=Endian.BIG,
        )
        raw = [decoder.decode_32bit_int() for _ in range(4)]
        returned_force = [raw[i] * self.ratio[i] for i in range(4)]
        return returned_force
    

    def read_force_plot(self, total_time=10, instruction="Reading forces — keep contact steady."):
        print(instruction)
        testkey = input(f"Recording for {total_time} seconds...")
        if testkey == 'q':
            return
        
        readings = [[] for _ in range(4)]
        timestamps = []
        t_start = time.time()

        while time.time() - t_start < total_time:
            forces = self.read_forces()
            if forces is not None:
                timestamps.append(time.time() - t_start)
                for i in range(4):
                    readings[i].append(forces[i])

        # Print stats
        print("\n--- Results ---")
        for i in range(4):
            arr = np.array(readings[i])
            print(f"CH{i+1}: mean={arr.mean():.4f}  sd={arr.std():.4f}")

        # Plot
        _, ax = plt.subplots(figsize=(10, 5))
        for i in range(4):
            ax.plot(timestamps, readings[i], label=f"CH{i+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force")
        ax.set_title("Force Sensor Readings")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    sensor = TouchForceSensorArray()
    # sensor.read_force_plot()
    try:
        while True:
            force = sensor.read_forces()
            if force is not None:
                print(f"CH1: {force[0]:8.2f} | CH2: {force[1]:8.2f} | "
                    f"CH3: {force[2]:8.2f} | CH4: {force[3]:8.2f}")
    except KeyboardInterrupt:
        pass