BEGIN {
  FS=","
  OFS=","
}
{
  if ($title_column != "" && $item_column != "") {
    print  $item_column, $title_column
  }
}
