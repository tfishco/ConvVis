var divHover = null,
    windowClick = false;

$(function(){
  $(window).mousedown(function(){
    windowClick = true;
  });

  $(window).mouseup(function(){
    windowClick = false;
  });

  $('.draggable').hover(function(){
    if(divHover === null){
      divHover = $(this);
    }
  }, function(){
    if(windowClick === false){
      divHover = null;
      $(this).css('z-index', '0');
    }
  });

  $(window).mousemove(function(e){
    if(windowClick === true && divHover != null){
      divHover.css({ top: e.clientY - divHover.height() / 2 + 'px', left: e.clientX - divHover.width() / 2 + 'px', position: 'absolute', zIndex: '1' });
    }
  });
})
