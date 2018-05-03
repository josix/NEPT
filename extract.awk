BEGIN {
  FS=","
  OFS=" "
}
{
  if ($user_column != "" && $item_column != "") {
    print $user_column, $item_column
  }
}
