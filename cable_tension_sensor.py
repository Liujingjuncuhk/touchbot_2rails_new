from pymodbus.client import ModbusSerialClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian


PORT = '/dev/touchbot_cableTensionSensor'
BAUDRATE = 115200
SLAVE_ID = 1
# Conversion ratios per channel (raw int -> physical units)
# RATIO = [1 / 7200, 1 / 11, 1 / 10000, 1 / 6900]
# RATIO = [1/720, 1/1100, 1/1000, 1/690]
RATIO = [1/100, 0.98/81, 0.98/53, 0.98/92]

class CableTensionSensor:
    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        slave_id: int = SLAVE_ID,
        ratio: list = RATIO,
    ):
        self.slave_id = slave_id
        self.ratio = ratio
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
        raw_ratioed = [raw[i] * self.ratio[i] for i in range(4)]
        returned_force = [round(raw_ratioed[1],3), round(raw_ratioed[0],3), round(raw_ratioed[3],3), round(raw_ratioed[2],3)]
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
        raw_ratioed = [raw[i] * self.ratio[i] for i in range(4)]
        returned_force = [round(raw_ratioed[1]-self.initial_offset[0],3), round(raw_ratioed[0]-self.initial_offset[1],3), round(raw_ratioed[3]-self.initial_offset[2],3), round(raw_ratioed[2]-self.initial_offset[3],3)]
        return returned_force

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    import time
    cable_tension_sensor = CableTensionSensor()
    total_read_time = 10
    start_time = time.time()
    count = 0
    while time.time() - start_time < total_read_time:
        forces = cable_tension_sensor.read_forces()
        if forces:
            count += 1
            print(f"CH1: {forces[0]:8.2f} | CH2: {forces[1]:8.2f} | "
                    f"CH3: {forces[2]:8.2f} | CH4: {forces[3]:8.2f}")
    cable_tension_sensor.close()
    print(f"Total reads: {count}")
    print(f"Hz: {count / total_read_time:.2f}")