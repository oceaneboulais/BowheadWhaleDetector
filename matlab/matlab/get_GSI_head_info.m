function head=get_GSI_head_info(head_info,year,Site,DASAR_ltr)



Iyear=find(contains(head_info.year_want,year));
Isite=find(contains(head_info.Site,Site));
Id=find(contains(head_info.strr,DASAR_ltr));

head=head_info.head{Iyear,Isite,Id};

 %head=head_info.head{Iyear,Isite,Id};
               

end
                