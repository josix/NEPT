BEGIN{
  FS=" "
  OFS=" "
}
{
  gsub(":.*", "", $5)
  if ($5 != "" && $1 != "") {
    if(count[$5" "$1] == ""){
      count[$5" "$1] = 1;
    }else{
      count[$5" "$1] = count[$5" "$1] + 1;
    }
  }
}
END{
  for (row in count){
    print row, count[row];
  }
}
