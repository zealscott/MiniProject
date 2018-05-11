<?php
/**
 * This file is part of workerman.
 *
 * Licensed under The MIT License
 * For full copyright and license information, please see the MIT-LICENSE.txt
 * Redistributions of files must retain the above copyright notice.
 *
 * @author walkor<walkor@workerman.net>
 * @copyright walkor<walkor@workerman.net>
 * @link http://www.workerman.net/
 * @license http://www.opensource.org/licenses/mit-license.php MIT License
 */

/**
 * 用于检测业务代码死循环或者长时间阻塞等问题
 * 如果发现业务卡死，可以将下面declare打开（去掉//注释），并执行php start.php reload
 * 然后观察一段时间workerman.log看是否有process_timeout异常
 */
//declare(ticks=1);

use \GatewayWorker\Lib\Gateway;

//包含数据库连接文件  
include('conn.php');

/**
 * 主逻辑
 * 主要是处理 onConnect onMessage onClose 三个方法
 * onConnect 和 onClose 如果不需要可以不用实现并删除
 */


$online = 0;
/*设置北京时区 */
date_default_timezone_set('PRC');

/* 将用户名，时间和发言内容进行插入 */
function update($content, $name)
{
    $link = mysqli_connect('localhost', 'duyuntao', '', 'mysql');
    $now = date("Y-m-d G:i:s");
    $str = "insert into massage (time, fromuser, content) values('$now', '$name', '$content')";
    mysqli_query($link, $str);  
}

/* 查找在线用户数量 */
function SearchOnline()
{
    $link = mysqli_connect('localhost', 'duyuntao', '', 'mysql');
    global $online;
    $str = "select count(*) from  online  where online.is_online = 1";
    $query = mysqli_query($link, $str);  
    if(mysqli_num_rows($query)){
        $count=mysqli_fetch_array($query)[0];
     }else{
         $count=0;
     }
     $online = $count;
     return;
}

/* 删除在线用户 */
function DeleteOnline($name)
{
    $link = mysqli_connect('localhost', 'duyuntao', '', 'mysql');
    $str = "delete from online where  username = '$name'";
    mysqli_query($link, $str);  
}

/* 业务逻辑 */
class Events
{
    /**
     * 当客户端连接时触发
     * 如果业务不需此回调可以删除onConnect
     * 
     * @param int $client_id 连接id
     */
    public static function onConnect($client_id)
    {
        // 向当前client_id发送数据 
       // Gateway::sendToClient($client_id, "$client_id");
    //   echo "login $client_id \n";
        // 向所有人发送
       // Gateway::sendToAll("$client_id login\r\n");
    }
    
   /**
    * 当客户端发来消息时触发
    * @param int $client_id 连接id
    * @param mixed $message 具体消息
    */
   public static function onMessage($client_id, $message)
   {
        global $online;
        $data = json_decode($message,true);
        switch ($data['type']) {
            case 'login':
                /* 登入消息 */
                SearchOnline();
                $new_message = array('type' => 'login', 'username' => $data['name'],'online'=>$online);
                Gateway::sendToAll(json_encode($new_message));
                break;
            case 'msg':
                /* 收到消息 */
                update($data['chatContent'], $data['name']);
                $new_message = array('type' => 'msg', 'username' => $data['name'], 'img' => $data['img'], 'msg' => $data['chatContent']);
                Gateway::sendToAll(json_encode($new_message));
                break;
            case 'logout':
                /* 删除Online名单 */ 
                DeleteOnline($data['name']);
                $online--;
                $new_message = array('type' => 'logout', 'username' => $data['name'],'online'=>$online);
                Gateway::sendToAll(json_encode($new_message));
                break;
            case 'file':
                /* 文件消息 */
                $new_message = array('type' => 'file', 'username' => $data['name'],'filename' => $data['filename']);
                Gateway::sendToAll(json_encode($new_message));
                break;
            default:
                break;
        }
   }
   
   /**
    * 当用户断开连接时触发
    * @param int $client_id 连接id
    */
   public static function onClose($client_id)
   {
    //    echo "logout $client_id\n";
    //    $result = array("type"=>"Logout");
    //    Gateway::sendToGroup($_SESSION['room'],json_encode($result));
       // 向所有人发送 
       //GateWay::sendToAll("$client_id logout\r\n");
   }
}
