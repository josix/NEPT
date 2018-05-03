BEGIN {
  FS=","
  OFS=" "
}
{
  print $user_column, $item_column
}
