create table train_test
as
select session_id, DateTime, user_id, product, campaign_id, webpage_id, product_category_1, product_category_2, user_group_id, gender, age_level, user_depth, city_development_index, var_1, is_click, 'train' src from train
union
select session_id, DateTime, user_id, product, campaign_id, webpage_id, product_category_1, product_category_2, user_group_id, gender, age_level, user_depth, city_development_index, var_1, null is_click, 'test' src from test
------------------------------------------------------------------------------------------
select count(1) from train_test; -- 592149
select count(1) from train_test where src = 'train'; -- 463291
select count(1) from train_test where src = 'test'; -- 128858
------------------------------------------------------------------------------------------
select count(1) from train_test where user_group_id is null; -- 23927

merge into train_test tr
using
(
	select distinct product, product_category_1, user_group_id from
	(
		select product, product_category_1, user_group_id, rank() over(partition by product, product_category_1 order by cnt desc) rnk
		from
			(select product, product_category_1, user_group_id, count(1) over(partition by product, product_category_1, user_group_id) cnt
				from train_test)
	)
	where rnk = 1
) tp on (tr.product = tp.product and tr.product_category_1 = tp.product_category_1)
when matched then
	update set tr.user_group_id = tp.user_group_id
	where tr.user_group_id is null
------------------------------------------------------------------------------------------
select count(1) from train_test where gender is null; -- 23927

merge into train_test tr
using
(
	select distinct product, product_category_1, user_group_id, gender from
	(
		select product, product_category_1, user_group_id, gender, rank() over(partition by product, product_category_1, user_group_id order by cnt desc) rnk
		from
			(select product, product_category_1, user_group_id, gender, count(1) over(partition by product, product_category_1, user_group_id, gender) cnt
				from train_test)
	)
	where rnk = 1
) tp on (tr.product = tp.product and tr.product_category_1 = tp.product_category_1 and tr.user_group_id = tp.user_group_id)
when matched then
	update set tr.gender = tp.gender
	where tr.gender is null
------------------------------------------------------------------------------------------
select count(1) from train_test where age_level is null; -- 23927

merge into train_test tr
using
(
	select distinct product, product_category_1, user_group_id, gender, age_level from
	(
		select product, product_category_1, user_group_id, gender, age_level, rank() over(partition by product, product_category_1, user_group_id, gender order by cnt desc) rnk
		from
			(select product, product_category_1, user_group_id, gender, age_level, count(1) over(partition by product, product_category_1, user_group_id, gender, age_level) cnt
				from train_test)
	)
	where rnk = 1
) tp on (tr.product = tp.product and tr.product_category_1 = tp.product_category_1 and tr.user_group_id = tp.user_group_id and tr.gender = tp.gender)
when matched then
	update set tr.age_level = tp.age_level
	where tr.age_level is null
------------------------------------------------------------------------------------------
select count(1) from train_test where user_depth is null; -- 23927

merge into train_test tr
using
(
	select distinct product, product_category_1, user_group_id, age_level, user_depth from
	(
		select product, product_category_1, user_group_id, age_level, user_depth, rank() over(partition by product, product_category_1, user_group_id, age_level order by cnt desc) rnk
		from
			(select product, product_category_1, user_group_id, age_level, user_depth, count(1) over(partition by product, product_category_1, user_group_id, age_level, user_depth) cnt
				from train_test)
	)
	where rnk = 1
) tp on (tr.product = tp.product and tr.product_category_1 = tp.product_category_1 and tr.user_group_id = tp.user_group_id and tr.age_level = tp.age_level)
when matched then
	update set tr.user_depth = tp.user_depth
	where tr.user_depth is null
------------------------------------------------------------------------------------------
select count(1) from train_test where city_development_index is null; -- 159738

merge into train_test tr
using
(
	select distinct product, product_category_1, user_group_id, city_development_index from
	(
		select product, product_category_1, user_group_id, city_development_index, rank() over(partition by product, product_category_1, user_group_id order by cnt desc, city_development_index) rnk
		from
			(select product, product_category_1, user_group_id, city_development_index, count(1) over(partition by product, product_category_1, user_group_id, city_development_index) cnt
				from train_test)
	)
	where rnk = 1
) tp on (tr.product = tp.product and tr.product_category_1 = tp.product_category_1 and tr.user_group_id = tp.user_group_id)
when matched then
	update set tr.city_development_index = tp.city_development_index
	where tr.city_development_index is null

merge into train_test tr
using
(
	select distinct product, product_category_1, city_development_index from
	(
		select product, product_category_1, city_development_index, rank() over(partition by product, product_category_1 order by cnt desc, city_development_index) rnk
		from
			(select product, product_category_1, city_development_index, count(1) over(partition by product, product_category_1, city_development_index) cnt
				from train_test)
	)
	where rnk = 1
) tp on (tr.product = tp.product and tr.product_category_1 = tp.product_category_1)
when matched then
	update set tr.city_development_index = tp.city_development_index
	where tr.city_development_index is null;

merge into train_test tr
using
(
	select distinct age_level, city_development_index from
	(
		select age_level, city_development_index, rank() over(partition by age_level order by cnt desc, city_development_index) rnk
		from
			(select age_level, city_development_index, count(1) over(partition by age_level, city_development_index) cnt
				from train_test)
	)
	where rnk = 1
) tp on (tr.age_level = tp.age_level)
when matched then
	update set tr.city_development_index = tp.city_development_index
	where tr.city_development_index is null
------------------------------------------------------------------------------------------
update train_test
set click_duration =
case
	when to_char(to_date(datetime, 'YYYY-MM-DD HH24:MI'), 'HH24') in ('18', '19', '20', '21') then '4'
	when to_char(to_date(datetime, 'YYYY-MM-DD HH24:MI'), 'HH24') in ('10', '11', '12', '13', '14', '15', '16', '17') then '3'
	when to_char(to_date(datetime, 'YYYY-MM-DD HH24:MI'), 'HH24') in ('07', '08', '09') then '2'
	when to_char(to_date(datetime, 'YYYY-MM-DD HH24:MI'), 'HH24') in ('22', '23', '00', '01', '02', '03', '04', '05', '06') then '1'
end
------------------------------------------------------------------------------------------
merge into train_test tr
using
(
	select tt.user_group_id, count(1) cnt
	from train_test tt join hist_user_logs h on (tt.user_id = h.user_id)
	where tt.is_click = 1
	group by tt.user_group_id
) tp on (tr.user_group_id = tp.user_group_id)
when matched then
	update set tr.click_cnt = tp.cnt
------------------------------------------------------------------------------------------