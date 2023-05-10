$(document).ready(function(){
    const ip = document.getElementById("ip").textContent.concat(":9099/admin");
    const socket = io.connect(ip);
    socket.on('result1', function(msg) {
        $('#log1').html(msg.dens);
        const image_element1 = document.getElementById('image1');
        image_element1.src="data:image/jpeg;base64,"+msg.obs;
    });
    socket.on('result2', function(msg) {
        $('#log2').html(msg.dens);
        const image_element2 = document.getElementById('image2');
        image_element2.src="data:image/jpeg;base64,"+msg.obs;
    });
});