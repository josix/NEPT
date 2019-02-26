{
  if(items[$1] == "") {
    items[$1] = $2;
  } else {
    items[$1] = items[$1]" "$2;
  }
}
END {
  for (user in items){
    print(user, items[user]);
  }
}
