import socket
import json
import time

class ServerSocket():
    def __init__(self, port, ip, mode, num=5):
        self.port = port
        self.ip = ip
        self.mode = mode
        self.num = num
        self.server_socket = ""
        if mode == 'tcp':
            self._build_tcp_conn()
        elif mode == 'udp':
            self._build_udp_conn()

    def _build_tcp_conn(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(self.num)

    def _build_udp_conn(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind((self.ip, self.port))


    def listen(self):
        if self.mode == 'tcp':
            conn, addr = self.server_socket.accept()
            while True:
                raw_data = self.tcp_handle_read(conn)
                if raw_data == '':
                    break
                print(raw_data)
                self.tcp_handle_send(conn, addr, '23333')

        elif self.mode == 'udp':
            # 单次监听，循环逻辑在外层实现
            raw_data, addr = self.udp_handle_read()
            # raw_data = json.loads(raw_data.decode('utf-8'))
            # msg = {
            #     'name':raw_data['name'],
            #     'message':'hello go'
            # }
            # self.udp_handle_send(addr, msg)
            return json.loads(raw_data.decode('utf-8')), addr
            #一直监听但是数据很难传输
            # while True:
            #     raw_data, addr = self.udp_handle_read()
            #     print(json.loads(raw_data.decode('utf-8')))
                # while True:
                #     name = input()
                #     self.udp_handle_send(('127.0.0.1',10001), name)
                    # self.udp_handle_send(addr, name)


    def udp_handle_send(self, addr, message):
        self.server_socket.sendto(str(message).encode(encoding='utf-8'), addr)

    def udp_handle_read(self):
        raw_data, addr = self.server_socket.recvfrom(1024)
        return raw_data, addr

    def tcp_handle_send(self, conn, addr, message):
        conn.sendto(str(message).encode(encoding='utf-8'), addr)

    def tcp_handle_read(self, conn):
        raw_data = conn.recv(1024).decode('utf-8')
        return raw_data

    def __del__(self):
        self.server_socket.close()


if __name__ == '__main__':
    server = ServerSocket(8888,'127.0.0.1', 'udp')
    server.listen()