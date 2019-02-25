BEGIN{
  FS=","
  OFS=" "
}
{
  if(count[$1" "$2] == ""){
    count[$1" "$2] = 1;
  }else{
    count[$1" "$2] = count[$1" "$2] + 1;
  }
}
END{
  for (row in count){
    print row, count[row];
  }
}

