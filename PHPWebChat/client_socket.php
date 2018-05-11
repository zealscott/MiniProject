<?php
session_start();
?>

/* JavaScript */
<script>

var fname = '';
// 获取聊天内容框
var chatContent = document.getElementsByClassName('chat-content')[0];

// 获取聊天输入框
var editBox = document.getElementsByClassName('edit-box')[0];

// 获取上传文件按钮
var fileButton = document.getElementsByClassName('upload-button')[0];

// 获取聊天输入框发送按钮
var editButton = document.getElementsByClassName('edit-button')[0];

// 获取用户名栏
var userName = document.getElementsByClassName('user-name')[0];

// 获取在线人数栏
var onlineCount = document.getElementsByClassName('online-count')[0];

// 把登录页面的名称放在右侧
userName.innerHTML = '<?php echo $_SESSION['username'];?>';
var userImg = document.getElementsByClassName('user-img')[0];

// 把登录页面的头像放在右侧
userImg.src = 'img/' + '<?php echo $_SESSION['img'];?>' + '.png';
var logOut = document.getElementsByClassName('log-out')[0];

// 发送按钮绑定点击发送事件
editButton.addEventListener('click', sendMessage);

// 发送按钮绑定点击上传事件
fileButton.addEventListener('click', uploadFile);

// 登出按钮绑定点击关闭页面
logOut.addEventListener('click', closePage);

// 绑定Enter键和发送事件
document.onkeydown = function (event) {
    var e = event || window.event;
    if (e && e.keyCode === 13) {
        if (editBox.value !== '') {
            editButton.click();
        }
    }
};


// socket部分
// onmessage 用来监听一个事件


// 当消息传来时进行的操作
function onmessage(e) {
    var data = JSON.parse(e.data);;
    switch(data['type']){
        case 'msg':
            var who = data['username'];
            var msg = data['msg'];
            var img = data['img'];
            if (who !== userName.textContent) {
            createOtherMessage(msg,img,who);
            }else{
                createMyMessage(msg);
            }
            break;
        case 'logout':
            onlineCount.innerHTML = 'Online:' + data['online'];
            createLogout(data['username']);
            break;
        case 'login':
            onlineCount.innerHTML = 'Online:' + data['online'];
            createLogin(data['username']);
            break;
        case 'file':
            createFile(data['username'],data['filename']);
            break;
    }
}

/* 新建一个websocket */
// var ws  = new WebSocket("ws://127.0.0.1:1234");
var ws  = new WebSocket("ws://10.11.1.43:8283");

//发送登录信息
ws.onopen = function () {
    var data = '{"type":"login","name":"' + userName.textContent + '"}';
    ws.send(data);
};

/* 绑定函数 */
ws.onmessage = onmessage;

// 登出
function closePage() {  
    /* 发送登出消息 */
    var data = '{"type":"logout","name":"' + userName.textContent + '"}';
    ws.send(data);

    /* 登出选项 */
    var userAgent = navigator.userAgent;
    if (userAgent.indexOf("Firefox") != -1 || userAgent.indexOf("Chrome") != -1) {
        window.location.href = "about:blank";
    } else {
        window.opener = null;
        window.open("", "_self");
        window.close();
    }
}


/* 将上传文件名传给所有用户 */
function uploadFile(){
    var data = '{"type":"file","name":"' + userName.textContent + '","filename":"' + fname + '"}';
    ws.send(data);

}


// 发送本机的消息
function sendMessage() {
    if (editBox.value != '') {
        // 发送
        var data = '{"type": "msg", "name":"' + userName.textContent + '","chatContent":"' + editBox.value + '","img":"' + userImg.src + '"}';
        ws.send(data);
        // 清空
        editBox.value = '';
    }
};

// 生成本机的聊天气泡
function createMyMessage(information) {
    var myMessageBox = document.createElement('div');
    myMessageBox.className = 'my-message-box';

    var messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    var text = document.createElement('span');
    text.innerHTML = information;
    messageContent.appendChild(text);
    myMessageBox.appendChild(messageContent);

    var arrow = document.createElement('div')
    arrow.className = 'message-arrow';
    myMessageBox.appendChild(arrow);

    var userInformation = document.createElement('div');
    userInformation.className = 'user-information';
    var userChatImg = document.createElement('img');
    userChatImg.className = 'user-chat-img';
    userChatImg.src = userImg.src;
    var userChatName = document.createElement('div');
    userChatName.className = 'user-chat-name';
    userChatName.innerHTML = userName.textContent;
    userInformation.appendChild(userChatImg);
    userInformation.appendChild(userChatName);
    myMessageBox.appendChild(userInformation);

    chatContent.appendChild(myMessageBox);

    chatContent.scrollTop = chatContent.scrollHeight;
}


// 生成其他用户的聊天气泡
function createOtherMessage(information,img,who) {
    var otherMessageBox = document.createElement('div');
    otherMessageBox.className = 'other-message-box';

    var otherUserInformation = document.createElement('div');
    otherUserInformation.className = 'other-user-information';
    var userChatImg = document.createElement('img');
    userChatImg.className = 'user-chat-img';
    userChatImg.src = img;
    var userChatName = document.createElement('span');
    userChatName.className = 'user-chat-name';
    userChatName.innerHTML = who;
    otherUserInformation.appendChild(userChatImg);
    otherUserInformation.appendChild(userChatName);
    otherMessageBox.appendChild(otherUserInformation);

    var otherMessageArrow = document.createElement('div');
    otherMessageArrow.className = 'other-message-arrow';
    otherMessageBox.appendChild(otherMessageArrow);

    var otherMessageContent = document.createElement('div');
    otherMessageContent.className = 'other-message-content';
    var text = document.createElement('span');
    text.innerHTML = information;
    otherMessageContent.appendChild(text);
    otherMessageBox.appendChild(otherMessageContent);

    chatContent.appendChild(otherMessageBox);

    chatContent.scrollTop = chatContent.scrollHeight;
}

/* 登出消息 */
function createLogout(name){

    var information = name + ' logged out !';
    var logoutMessage = document.createElement('div');
    logoutMessage.className = 'log-box';
    var text = document.createElement('span');
    text.innerHTML = information;
    logoutMessage.appendChild(text);
    chatContent.appendChild(logoutMessage);
}

/* 登入消息 */
function createLogin(name){
    var information = name + ' logs in !';
    var logoutMessage = document.createElement('div');
    logoutMessage.className = 'log-box';
    var text = document.createElement('span');
    text.innerHTML = information;
    logoutMessage.appendChild(text);
    chatContent.appendChild(logoutMessage);
}

/* 上传文件消息 */
function createFile(name,filename){
    var information = name + ' upload file ' + filename;
    var logoutMessage = document.createElement('div');
    logoutMessage.className = 'log-box';
    var text = document.createElement('span');
    text.innerHTML = information;
    logoutMessage.appendChild(text);
    chatContent.appendChild(logoutMessage);
}

</script>