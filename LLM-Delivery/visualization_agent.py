import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPolygonF
from PyQt5.QtCore import Qt, QPointF, QRectF

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)  # Change to NoDrag to handle manually
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumSize(800, 600)

        # Initial scale factor
        self.scale(0.5, 0.5)
        self._last_mouse_pos = None  # Track the last mouse position

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last_mouse_pos is not None:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """处理鼠标滚轮事件，实现缩放功能"""
        zoom_factor = 1.15

        if event.angleDelta().y() > 0:
            # 放大
            self.scale(zoom_factor, zoom_factor)
        else:
            # 缩小
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

class CustomerItem(QGraphicsItem):
    def __init__(self, customer):
        super().__init__()
        self.customer = customer
        self.setPos(customer.x, customer.y)
        # 设置Z值使其显示在最上层
        self.setZValue(10)

    def boundingRect(self):
        return QRectF(-25, -25, 50, 50)

    def paint(self, painter, option, widget):
        # 使用更大、更明显的圆形
        painter.setPen(QPen(QColor(0, 0, 255), 3))
        painter.setBrush(QBrush(QColor(0, 0, 255, 200)))
        painter.drawEllipse(QRectF(-200, -200, 400, 400))

        # 添加文字标签
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(-200, -100, 400, 200), Qt.AlignCenter, "C")

class StoreItem(QGraphicsItem):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.setPos(store.x, store.y)
        # 设置Z值使其显示在最上层
        self.setZValue(10)
        
    def boundingRect(self):
        return QRectF(-25, -25, 50, 50)

    def paint(self, painter, option, widget):
        # 使用更大、更明显的方形
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.setBrush(QBrush(QColor(255, 0, 0, 200)))
        painter.drawRect(QRectF(-200, -200, 400, 400))
        
        # 添加文字标签
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(-200, -100, 400, 200), Qt.AlignCenter, "S")

class DeliveryManItem(QGraphicsItem):
    def __init__(self, delivery_man):
        super().__init__()
        self.delivery_man = delivery_man
        self.setPos(delivery_man.position.x, delivery_man.position.y)
        
    def boundingRect(self):
        return QRectF(-25, -25, 50, 50)

    def paint(self, painter, option, widget):
        # 使用更大、更明显的方形
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
        painter.drawRect(QRectF(-200, -200, 400, 400))
        
        # 添加文字标签
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(-200, -100, 400, 200), Qt.AlignCenter, "D")

class SupplyPointItem(QGraphicsItem):
    def __init__(self, supply_point):
        super().__init__()
        self.supply_point = supply_point
        self.setPos(supply_point.x, supply_point.y)
        self.setZValue(10)

    def boundingRect(self):
        return QRectF(-25, -25, 50, 50)

    def paint(self, painter, option, widget):
        # 使用更大、更明显的方形
        painter.setPen(QPen(QColor(0, 255, 0), 3))
        painter.setBrush(QBrush(QColor(0, 255, 0, 200)))
        painter.drawRect(QRectF(-200, -200, 400, 400))
        
        # 添加文字标签
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(-200, -100, 400, 200), Qt.AlignCenter, "P")

class RoadItem(QGraphicsItem):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
        
    def boundingRect(self):
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        
        # 确保边界矩形足够大以包含整条道路
        min_x = min(x1, x2) - 500
        min_y = min(y1, y2) - 500
        width = abs(x2 - x1) + 1000
        height = abs(y2 - y1) + 1000
        
        return QRectF(min_x, min_y, width, height)

    def paint(self, painter, option, widget):
        painter.setPen(QPen(QColor(50, 50, 50), 5))
        # 使用QPointF来处理浮点坐标
        painter.drawLine(
            QPointF(self.start.x, self.start.y),
            QPointF(self.end.x, self.end.y)
        )

class VisualizationWindow(QMainWindow):
    def __init__(self, llm_delivery):
        super().__init__()
        self.llm_delivery = llm_delivery.delivery_manager
        
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        
        self.setWindowTitle("LLM Delivery Visualization")
        self.setGeometry(100, 100, 1000, 800)
        
        self.initialize_scene()
        
    def initialize_scene(self):
        # 添加道路
        for road in self.llm_delivery.map.edges:
            start_node, end_node = road.node1.position, road.node2.position
            self.scene.addItem(RoadItem(start_node, end_node))
        
        
        # 打印客户位置信息以进行调试
        for i, customer in enumerate(self.llm_delivery.customers):
            # print(f"客户 {i}: 位置 ({customer.position.x}, {customer.position.y})")
            customer_item = CustomerItem(customer)
            self.scene.addItem(customer_item)
        
        # 打印商店位置信息以进行调试
        # print("商店位置:")
        for i, store in enumerate(self.llm_delivery.stores):
            # print(f"商店 {i}: 位置 ({store.position.x}, {store.position.y})")
            store_item = StoreItem(store)
            self.scene.addItem(store_item)
        
        # 添加配送员
        if self.llm_delivery.delivery_men:
            for delivery_man in self.llm_delivery.delivery_men:
                if hasattr(delivery_man, 'position'):  # 确保配送员有位置属性
                    self.scene.addItem(DeliveryManItem(delivery_man))
        
        # 添加供应点
        for supply_point in self.llm_delivery.supply_points:
            self.scene.addItem(SupplyPointItem(supply_point))
        
        # 设置场景边界，确保所有内容可见
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        
        # 调整视图以显示整个场景
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        # 重置变换以避免缩放过小
        self.view.resetTransform()
        # 使用更合适的初始缩放
        self.view.scale(0.01, 0.01)  # 使用较小的缩放因子
        
        # 打印一些调试信息
        # print(f"场景中的项目数量: {len(self.scene.items())}")
        # print(f"场景边界: {self.scene.sceneRect()}")
        # print(f"建筑物数量: {len(self.llm_delivery.buildings)}")
        # print(f"客户数量: {len(self.llm_delivery.customers)}")
        # print(f"商店数量: {len(self.llm_delivery.stores)}")
        # print(f"道路数量: {len(self.llm_delivery.map.edges)}")

def visualize_agents(llm_delivery):
    """
    使用PyQt5可视化LLMDelivery中的建筑物、客户和商店
    
    Args:
        llm_delivery: LLMDelivery实例
    """
    app = QApplication(sys.argv)
    window = VisualizationWindow(llm_delivery)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # 测试代码，如果直接运行此文件
    from LLMDelivery import LLMDelivery
    llm_delivery = LLMDelivery(2, 2, 2, "input")
    # print(llm_delivery.delivery_manager.map.edges)
    visualize_agents(llm_delivery)
