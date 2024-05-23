

class FrogLogger:
    def __init__(self, file_path):
        """初始化日志记录器，指定日志输出的文件位置并创建文件"""
        self.file_path = file_path
        # 打开文件，准备写入，如果文件不存在则创建，如果文件已存在则清空已有内容
        self.file = open(file_path, 'w')

    def write_log(self, message):
        """向指定的文件中写入日志信息"""
        self.file.write(message + '\n')

    def close(self):
        """保存并关闭日志文件"""
        self.file.close()