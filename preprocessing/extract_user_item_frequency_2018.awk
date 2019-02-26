BEGIN{
  FS=","
  OFS=" "
}
{
  if($15 !~ /2018/) next;
  if($5 == "")next;
  if(count[$5" "$11] == ""){
    count[$5" "$11] = 1;
  }else{
    count[$5" "$11] = count[$5" "$11] + 1;
  }
}
END{
  for (row in count){
    print row, count[row];
  }
}
