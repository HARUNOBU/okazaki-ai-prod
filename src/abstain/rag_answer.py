from abstain.tourism import should_abstain_tourism, TOURISM_ABSTAIN_MESSAGE

def answer_query(query: str):
    category = route_category(query)

    if category == "tourism":
        docs, best_dist = search_tourism(query)

        if should_abstain_tourism(
            query=query,
            best_dist=best_dist,
            dist_threshold=TOURISM_DIST_TH
        ):
            return {
                "answer": TOURISM_ABSTAIN_MESSAGE,
                "answered_by": "abstain",
                "category": "tourism"
            }

        return generate_tourism_answer(docs)

    # 他カテゴリは従来通り
