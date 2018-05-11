# 使用

本项目前端用WebSocket，后端使用[Workerman](http://www.workerman.net/)进行通讯。

## 代码逻辑

- `index.php`

  入口网站

- `client_socket.php`

  客户端socket、生成聊天气泡等

- `login.php`

  登录验证代码

- `upload_file.php`

  上传文件功能（待完善）

- `conn.php`

  基于MySQL的数据库配置。其中涉及到三个表：

  `online`表示在线用户的相关信息

  `massage`表示发送信息

  `reguser`为注册用户

- `main.php`

  聊天室主界面

- `..\GatewayWorker\Applications\YourApp\Events.php`

  服务端逻辑

## socket配置

在`..\GatewayWorker\Applications\YourApp`文件夹下：

- 修改服务注册地址 

  在`start_gateway.php`、`start_businessworker.php`，`start_register.php`中修改对应地址

- 修改socket端口

  在`start_gateway.php`中修改

  在`client_socket.php`保持一致

## 启动服务端

在`..\GatewayWorker\`目录下输入：

> php start.php start

出现如下界面说明服务端监听成功：

![1525939550084](http://wx1.sinaimg.cn/mw690/0060lm7Tly1fr6b2s6jaqj30iy04at8u.jpg)

## 打开聊天室

![1525939661040](http://wx4.sinaimg.cn/mw690/0060lm7Tly1fr6b4h2c4xj30ox0o8wk3.jpg)

![1525939697018](http://wx3.sinaimg.cn/mw690/0060lm7Tly1fr6b501j6kj30ng0mignm.jpg)



# 项目介绍

## 功能

一个聊天室需要有用户登录，上线提醒，发送消息，上传文件等多个功能。其中的难点是如何让前后端保持通讯并即使发送消息，因此分为前后端进行介绍。

## 前端

前段主要包含两个页面，分别是`index.php`和`main.php`。

业务逻辑主要在`client_socket.php`中实现

## 后端

1. 数据库

   采用MySQL数据库进行存储：

   `online`表示在线用户的相关信息（IP地址、用户名等）

   `massage`表示发送信息（时间、信息内容）

   `reguser`为注册用户（用户注册时间、密码、用户名）

2. Socket

   后端采用[Workerman](http://www.workerman.net/)的`GatewayWorker`类实现长连接功能。支持大量并发操作。

# 待完善

- [ ] 目前只完成了上传文件功能，但并没有实现将上传的文件下载。

- [ ] 实现多聊天室功能。

- [ ] 查看聊天历史功能。