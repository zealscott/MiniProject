<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>chat-room</title>
    <link rel="stylesheet" href="css/main.css">
</head>

<body>
    <div class="chat-box">
        <!--聊天框头部-->
        <div class="chat-header">
            <div class="button-box">
                <input type="button" class="log-out" value="LOGOUT">
            </div>
        </div>
        <!--聊天框主体-->
        <div class="chat-body">
            <!--聊天框左侧-->
            <div class="chat-body-left">
                <!--聊天框左侧聊天内容-->
                <div class="chat-content"></div>
                <!--聊天框左侧聊天输入框-->
                <div class="chat-edit">
                    <input type="text" class="edit-box" placeholder="Please Type You Message" maxlength="50"> 
                    <input type="button" class="edit-button" value="SEND">
                </div>

            </div>
            <!--聊天框右侧-->
            <div class="chat-body-right">
                <!--聊天框右侧统计人数-->
                <div class="online-count">Online:0</div>
                <!--聊天框右侧用户名-->
                <div class="user-name">user-name</div>
                <!--聊天框右侧头像-->
                <img class="user-img" />
                <!-- 实现上传文件 -->
                <form action="upload_file.php" method="post" enctype="multipart/form-data">
                <div style="margin:0 auto;width:100px;">
                <input type="file" name="file"  id="file" class = "file-button">
                <input type="submit" name="submit" value="提交"  class = "upload-button">
                </div>
                </form>
                <!-- 结束 -->
            </div>
        </div>
    </div>
</body>
<?php
include('client_socket.php');
?>

</html>