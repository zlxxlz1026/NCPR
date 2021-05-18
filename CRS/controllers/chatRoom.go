package controllers

import (
	"encoding/json"
	"fmt"
	"github.com/astaxie/beego"
	"github.com/gorilla/websocket"
	"net"
	_ "net/http"
	"sync"
)

type Client struct {
	conn *websocket.Conn	// 用户websocket连接
	name string				// 用户名称
	mu   *sync.Mutex 		//TODO:注意不是锁无效，而是range是拷贝，相当于每个协程拷贝一个锁的状态，所以应该用指针
	content chan Message
}

// 1.设置为公开属性(即首字母大写)，是因为属性值私有时，外包的函数无法使用或访问该属性值(如：json.Marshal())
// 2.`json:"name"` 是为了在对该结构类型进行json编码时，自定义该属性的名称
type Message struct {
	EventType byte	`json:"type"`		// 0表示用户发布消息；1表示用户进入；2表示用户退出
	Name string		`json:"name"`		// 用户名称
	Message string	`json:"message"`	// 消息
}

var (
	// 此处要设置有缓冲的通道。因为这是goroutine自己从通道中发送并接受数据。
	// 若是无缓冲的通道，该goroutine发送数据到通道后就被锁定，需要数据被接受后才能解锁，而恰恰接受数据的又只能是它自己
	join = make(chan Client, 10)			// 用户加入通道
	leave = make(chan Client, 10)			// 用户退出通道
	message = make(chan Message, 10)		// 消息通道
	clients = make(map [Client] bool)		// 用户映射
)

func init() {
	//TODO:协程创建协程级别一致吗？
	conn := BuildUdpConn(10001, "127.0.0.1", "127.0.0.1:8888")
	go listChan(conn)
	go listenServerMessage(conn)
}

//该函数可以用于整个聊天室，将服务器消息进行广播
func listenServerMessage(conn net.Conn) {
	for {
		//收到btyes数组，要让js解析需要生成需求的json格式.
		//广播消息
		//recvStr := UdpRecvHandler(conn)
		//单人消息
		recvStr, name:= UdpRecvHandler(conn)
		recvMsg := Message{byte(4), "推荐机器人", recvStr}
		robotData, err := json.Marshal(recvMsg)
		if err != nil {
			beego.Error("Fail to marshal message:", err)
			return
		}
		//群聊bot
		//broadcaster(robotData)
		//单人bot
		personal(name, robotData)
	}
}


//该函数的初衷是解决websocket不能并发写的问题，由于现在并发使用了锁控制，该函数暂时无用
func chanToWebSocket(c *Client) {
	for {
		msg := <- c.content
		robotData, err := json.Marshal(msg)
		if err != nil {
			beego.Error("Fail to marshal message:", err)
			return
		}
		if c.conn.WriteMessage(websocket.TextMessage, robotData) != nil {
			beego.Error("Fail to write message")
			return
		}
	}
}

//个人聊天
func personal(name string, data []byte) {
	for client := range clients {
		if name != client.name {
			continue
		}
		// 将数据编码成json形式，data是[]byte类型
		fmt.Println("=======the json message is", string(data))	// 转换成字符串类型便于查看
		client.mu.Lock()
		if client.conn.WriteMessage(websocket.TextMessage, data) != nil {
			beego.Error("Fail to write message")
		}
		client.mu.Unlock()
	}
}
// 广播，用于多人聊天
func broadcaster(data []byte) {
	for client := range clients {
		//client.content <- msg
		// 将数据编码成json形式，data是[]byte类型
		//json.Marshal()只会编码结构体中公开的属性(即大写字母开头的属性)
		fmt.Println("=======the json message is", string(data))	// 转换成字符串类型便于查看
		client.mu.Lock()
		if client.conn.WriteMessage(websocket.TextMessage, data) != nil {
			beego.Error("Fail to write message")
		}
		client.mu.Unlock()
	}
}

//监听通道信息
func listChan(conn net.Conn) {
	defer conn.Close()
	for {
		// 哪个case可以执行，则转入到该case。都不可执行，则堵塞。
		select {
		// 消息通道中有消息则执行，否则堵塞
		case msg := <-message:
			str := fmt.Sprintf("broadcaster-----------%s send message: %s", msg.Name, msg.Message)
			beego.Info(str)
			data, err := json.Marshal(msg)
			if err != nil {
				beego.Error("Fail to marshal message:", err)
				return
			}
			// 将某个用户发出的消息发送给所有用户
			//if int(msg.EventType) == 1 {
			//	continue
			//}

			//broadcaster(data)
			personal(msg.Name, data)
			UdpSendHandler(conn, string(data))

		// 有用户加入
		case client := <-join:
			//go sendToWebSocket(&client)
			str := fmt.Sprintf("broadcaster-----------%s join in the chat room\n", client.name)
			beego.Info(str)

			clients[client] = true	// 将用户加入映射
			//go HandleServerSend(conn, client)
			// 将用户加入消息放入消息通道
			var msg Message
			msg.Name = client.name
			msg.EventType = 1
			msg.Message = fmt.Sprintf("%s join in, there are %d preson in room", client.name, len(clients))

			// 此处要设置有缓冲的通道。因为这是goroutine自己从通道中发送并接受数据。
			// 若是无缓冲的通道，该goroutine发送数据到通道后就被锁定，需要数据被接受后才能解锁，而恰恰接受数据的又只能是它自己
			message <- msg

		// 有用户退出
		case client := <-leave:
			str := fmt.Sprintf("broadcaster-----------%s leave the chat room\n", client.name)
			beego.Info(str)

			// 如果该用户已经被删除
			if !clients[client] {
				beego.Info("the client had leaved, client's name:"+client.name)
				break
			}
			delete(clients, client)	// 将用户从映射中删除
			// 将用户退出消息放入消息通道
			var msg Message
			msg.Name = client.name
			msg.EventType = 2
			msg.Message = fmt.Sprintf("%s leave, there are %d preson in room", client.name, len(clients))
			message <- msg
		}
	}
}
