<?php
date_default_timezone_set('PRC'); 
//登录  
if (!isset($_POST['submit'])) {
    exit('非法访问!');
}
// get _POST data
$username = $_POST['username'];
$password = $_POST['password'];
if (!$_POST['img'])
/* 若默认图片 */
    $img = 3;
else 
    $img = $_POST['img'];


$furl=getenv("HTTP_REFERER");
$ip=getenv("REMOTE_ADDR");
$now=date("Y-m-d G:i:s");

//包含数据库连接文件  
include('conn.php');
//检测用户名及密码是否正确  
$check_name = mysqli_query($link, "select * from reguser where username='$username'");
$result = mysqli_num_rows($check_name);

if ($result == 0) {
    /* 若用户名不存在 */
    mysqli_query($link, "insert into reguser (username,password,regtime)values('$username','$password','$now')");

    session_start();
    $_SESSION['username'] = $username;
    $_SESSION['userid'] = $result['userid'];
    $_SESSION['img'] = $img;

    $queryA="insert into online (username,intime,ip,is_online)values('$username','$now','$ip','1')";
    $resultA=mysqli_query($link,$queryA);
    // echo "<script>alert(\"username:".$_SESSION['username']."\");</script>";
    echo "<script language=javascript>alert('欢迎你！');location.href='main.php';</script>";

} else {
    if ($result > 1) {
        //用户名已经存在
        echo "<script language=javascript>alert('该用户已经存在,请选择其它的用户名,谢谢!');history.back();</script>";

    } else if ($result = 1) {
        $check_pw = mysqli_query($link, "select * from reguser where username='$username' and password = '$password'");
        $result2 = mysqli_num_rows($check_pw);
        if ($result2 > 0) {
            /* 用户名和密码都正确 */
            session_start();
            $_SESSION['username'] = $username;
            $_SESSION['userid'] = $result['userid'];
            $_SESSION['img'] = $img;
            $queryA="insert into online (username,intime,ip,is_online)values('$username','$now','$ip','1')";
            $resultA=mysqli_query($link,$queryA);
            echo "<script language=javascript>alert('欢迎你！');location.href='main.php';</script>";
            exit;
        } else {
            echo "<script language=javascript>alert('用户名或密码错误！');history.back();</script>";
        }
    }
}

?>  