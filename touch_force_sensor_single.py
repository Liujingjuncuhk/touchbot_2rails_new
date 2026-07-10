from pymodbus.client import ModbusSerialClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian

PORT = '/dev/touchbot_touchForceSensor'
BAUDRATE = 115200
SLAVE_ID = 1
RATIO = 0.01


class TouchForceSensor:
    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        slave_id: int = SLAVE_ID,
        ratio: float = RATIO,
    ):
        self.slave_id = slave_id
        self.ratio = ratio
        self._client = ModbusSerialClient(port=port, baudrate=baudrate, timeout=0.2)
        if not self._client.connect():
            raise RuntimeError(f"Cannot open serial port {port}")

    def read_force(self) -> float:
        """Read the current single-channel force value.

        Returns:
            Force in physical units (float), or None if the read fails.
        """
        result = self._client.read_holding_registers(
            address=80, count=2, slave=self.slave_id
        )
        if result.isError():
            return None

        decoder = BinaryPayloadDecoder.fromRegisters(
            result.registers,
            byteorder=Endian.BIG,
            wordorder=Endian.BIG,
        )
        return decoder.decode_32bit_int() * self.ratio

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    with TouchForceSensor() as sensor:
        try:
            while True:
                force = sensor.read_force()
                if force is not None:
                    print(f"Force: {force:8.2f}")
        except KeyboardInterrupt:
            pass