<?php
session_start();  //开启session
?>

	<!DOCTYPE html>
	<html>
	<head>
		<title>欢迎进入聊天室</title>
		<link href="css/index.css" rel='stylesheet' type='text/css' />
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
		<meta name="keywords" content="App Sign in Form,Login Forms,Sign up Forms,Registration Forms,News latter Forms,Elements"
		 ./>
		<script type="application/x-javascript"> addEventListener("load", function() { setTimeout(hideURLbar, 0); }, false); function hideURLbar(){ window.scrollTo(0,1); } </script>
		</script>
	</head>

	<body>
		<h1>登录聊天室</h1>
		<div class="app-cam">
			<div class="cam">
				<class="img-responsive" alt="" />
			</div>

			<!--头像栏-->
			<div class="picture-carousel">
				<div class="arrow left-arrow">
					<div class="before-arrow"></div>
				</div>
				<img class="p1 img-setting" src="img/1.png" alt="1.png">
				<img class="p2 img-setting" src="img/2.png" alt="2.png">
				<img class="p3 img-setting" src="img/3.png" alt="3.png">
				<img class="p2 img-setting" src="img/4.png" alt="4.png">
				<img class="p1 img-setting" src="img/5.png" alt="5.png">
				<div class="arrow right-arrow">
					<div class="after-arrow"></div>
				</div>
			</div>

			<form action="login.php" method="post">
				<input type="text" class="text" name="username" id="username" onfocus="this.value = '';" onblur="if (this.value == '') {this.value = 'username';}">
				<input type="password" name="password" id="password" onfocus="this.value = '';" onblur="if (this.value == '') {this.value = 'password';}">

				<input type="hidden" id="img" name="img" value="3" />

				<div class="submit">
					<input type="submit" name="submit" value="登录/注册">
				</div>
				<div class="new">
					<p>
						<a href="#">忘记密码</a>
					</p>
					<p class="sign">
						<a href="login.php?action=logout"> 注销</a>
					</p>
					<div class="clear"></div>
				</div>
			</form>
		</div>
		<!--start-copyright-->
		<div class="copy-right">
			<p>Copyright &copy; scottdu</p>
		</div>
		<!--//end-copyright-->
	</body>
	<script src="js/login.js"></script>

	</html>