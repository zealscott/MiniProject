//用于存储图片顺序
var imgArray = ['1', '2', '3', '4', '5']; 



//获取箭头
var leftArrow = document.getElementsByClassName('left-arrow')[0];
var rightArrow = document.getElementsByClassName('right-arrow')[0];

// 添加左箭头监听事件
leftArrow.addEventListener('click', function () {
    imgArray.unshift(imgArray[imgArray.length - 1]); //把最后的元素放在第一位
    imgArray.pop();
    carouselImg();

    document.getElementById("img").value = imgArray[2];
});

// 添加右箭头监听事件
rightArrow.addEventListener('click', function () {
    imgArray.push(imgArray[0]); //把第一个元素放在最后
    imgArray.shift();
    carouselImg();

    document.getElementById("img").value = imgArray[2];
});

// 切换图片
function carouselImg() {
    for (var count = 0; count < imgArray.length; count++) {
        document.getElementsByTagName('img')[count].src = 'img/' + imgArray[count] + '.png';
        document.getElementsByTagName('img')[count].alt = imgArray[count] + '.png';
    };
};


