"""Small ROS helpers shared by the independently runnable nodes."""
import math
import time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor


def sensor_qos(depth=1, reliable=True):
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth,
                      reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT)


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class MovingAverageMetrics:
    """Allocation-light exponential moving averages for runtime diagnostics."""
    def __init__(self, alpha=.1):
        if not 0 < alpha <= 1:
            raise ValueError('Moving-average alpha must be in (0,1]')
        self.alpha, self.values = alpha, {}

    def update(self, **measurements):
        for name, value in measurements.items():
            value = float(value)
            previous = self.values.get(name, value)
            self.values[name] = previous + self.alpha * (value - previous)
        return {f'{name}_average': value for name, value in self.values.items()}


class PipelineNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.status_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.last_input = time.monotonic()
        self.last_status = None
        self.reliable_transport = self.param('reliable_transport', True)

    def image_qos(self, depth=1):
        return sensor_qos(depth, self.reliable_transport)

    def param(self, name, default):
        return self.declare_parameter(name, default, ParameterDescriptor(read_only=True)).value

    def positive(self, name, default):
        value = float(self.param(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    def status(self, reason, level=DiagnosticStatus.OK, **values):
        # diagnostic_msgs.level is ROS `byte`, represented by one Python byte
        # (not uint8/int); its C serializer otherwise aborts the process.
        if isinstance(level, int):
            level = bytes([level])
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        item = DiagnosticStatus(name=self.get_name(), hardware_id='object_pose_pipeline',
                                level=level, message=reason)
        item.values = [KeyValue(key=str(k), value=str(v)) for k, v in values.items()]
        msg.status = [item]
        self.status_pub.publish(msg)
        if reason != self.last_status:
            self.get_logger().info(reason)
            self.last_status = reason

    def age(self, header):
        return (self.get_clock().now().nanoseconds - stamp_ns(header.stamp)) / 1e9


def run(factory, args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = None
    try:
        node = factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
