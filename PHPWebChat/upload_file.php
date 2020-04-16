<?php

if ($_FILES["file"]["error"] > 0 || $_FILES["file"]["size"]/1024 > 2000)
{
    echo "<script language=javascript>alert('error!');location.href='main.php';</script>";
}
else
{
// echo "Upload: " . $_FILES["file"]["name"] . "<br />";
// echo "Type: " . $_FILES["file"]["type"] . "<br />";
// echo "Size: " . ($_FILES["file"]["size"] / 1024) . " Kb<br />";
// echo "Temp file: " . $_FILES["file"]["tmp_name"] . "<br />";

if (file_exists("upload/" . $_FILES["file"]["name"]))
    {
        echo "<script language=javascript>alert('file already existes!');location.href='main.php';</script>";
    // echo $_FILES["file"]["name"] . " already exists. ";
    }
else
    {
    move_uploaded_file($_FILES["file"]["tmp_name"],"upload\\" . $_FILES["file"]["name"]);
    // echo "Stored in: " . "upload\\" . $_FILES["file"]["name"];
    echo "<script language=javascript>alert('file  uploaded sucess !');location.href='main.php';</script>";
    }
}

?>