package controllers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
)

func checkError(err error){
	if  err != nil {
		fmt.Println("Error: %s", err.Error())
		os.Exit(1)
	}
}

//建立短tcp短连接
//TODO: 注意client关闭tcp连接会有time_wait阶段（协议规定）
func DialTcp(port int, ip, addr, message string) {
	netAddr := &net.TCPAddr{
		IP:   net.ParseIP(ip),
		Port: port,
		Zone: "",
	}
	fmt.Println("DialTcp addr is: ", netAddr)
	d := net.Dialer{
		LocalAddr: netAddr,
	}
	conn, err := d.Dial("tcp", addr)
	checkError(err)
	defer conn.Close()
	_, err = conn.Write([]byte(message))
	checkError(err)
	buffer := make([]byte, 1024)
	_, err = conn.Read(buffer)
	//buffer, err := ioutil.ReadAll(conn)
	checkError(err)
	fmt.Println(string(buffer))
}

//建立udp短连接并发送消息
func DialUdp(port int, ip, addr, message string) {
	netAddr := &net.UDPAddr{
		IP:   net.ParseIP(ip),
		Port: port,
		Zone: "",
	}
	fmt.Println("DialTcp addr is: ",netAddr)
	d := net.Dialer{
		LocalAddr:netAddr,
	}
	conn, err := d.Dial("udp", addr)
	defer conn.Close()
	checkError(err)
	input := bufio.NewScanner(os.Stdin)
	for {
		input.Scan()
		name := input.Text()
		_, err = conn.Write([]byte(name))
		checkError(err)
		buffer := make([]byte, 1024)
		n, err := conn.Read(buffer)
		checkError(err)
		fmt.Println(string(buffer[0:n]))
	}
}

//相当于建立udp长连接
func BuildUdpConn(port int, ip, addr string) (net.Conn) {
	netAddr := &net.UDPAddr{
		IP:   net.ParseIP(ip),
		Port: port,
		Zone: "",
	}
	fmt.Println("DialTcp addr is: ",netAddr)
	d := net.Dialer{
		LocalAddr:netAddr,
	}
	conn, err := d.Dial("udp", addr)
	checkError(err)
	return conn
}

func UdpSendHandler(conn net.Conn, message string) {
	_, err := conn.Write([]byte(message))
	checkError(err)
}

func UdpRecvHandler(conn net.Conn) (string, string) {
	buffer := make([]byte, 2048)
	n, err := conn.Read(buffer)
	checkError(err)
	//单人
	m := BytesToMap(buffer[:n])
	return m["msg"], m["name"]
	//广播
	//return string(buffer[:n])
}

func BytesToMap(bytesStr []byte) (map[string]string) {
	m := make(map[string]string)
	//fmt.Println(string(bytesStr))
	err := json.Unmarshal(bytesStr, &m)
	checkError(err)
	return m
}

func main() {
	//netAddr := &net.TCPAddr{Port:10001}
	//netAddr.IP = net.ParseIP("127.0.0.1")
	//fmt.Println(netAddr)
	//d := net.Dialer{LocalAddr:netAddr}
	//conn, err := d.Dial("tcp", "127.0.0.1:8888")
	//checkError(err)
	//defer conn.Close()
	//fmt.Println(conn.LocalAddr())
	//fmt.Println(conn.RemoteAddr())
	//conn.Write([]byte("Hello world!"))
	//x := make([]byte, 1024)
	//conn.Read(x)
	//fmt.Println("qqqqq ", string(x))
	//fmt.Println("send msg")
	//DialTcp(10001, "127.0.0.1", "127.0.0.1:8888", "hello python")
	DialUdp(10001, "127.0.0.1", "127.0.0.1:8888", "hello python")
}
