{
  if(access_times[$1] == ""){
    access_times[$1] = 1;
    access_items[$1] = "u"$0;
  } else{
    access_times[$1] = access_times[$1] + $3;
    access_items[$1] = "u"$0"\n"access_items[$1];
  }
}

END {
  for (user in access_times) {
    if (access_times[user] < 5) continue;
    print(access_items[user]);
  }
}
